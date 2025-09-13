from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from ..models import db, Product, Investment, SoilReport, Transaction, GovScheme, SchemeApplication, User
from datetime import datetime, timedelta
from functools import wraps
from sqlalchemy import func
from ..ml_models import CropPredictionModel, YieldForecastingModel, SoilAnalysisModel
import pandas as pd
import numpy as np

farmer_bp = Blueprint('farmer', __name__)

def farmer_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('auth.login', role='farmer'))
        if current_user.role != 'farmer':
            flash('Access denied. Farmer privileges required.', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

@farmer_bp.route('/farmer/dashboard')
@login_required
@farmer_required
def dashboard():
    try:
        # Get pending investment requests
        pending_requests = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='seeking'
        ).order_by(Investment.created_at.desc()).all()
        
        # Get active investments
        active_investments = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='active'
        ).order_by(Investment.created_at.desc()).all()
        
        # Get completed investments
        completed_investments = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='completed'
        ).order_by(Investment.created_at.desc()).all()
        
        # Get pending offers
        pending_offers = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='offered'
        ).order_by(Investment.created_at.desc()).all()
        
        # Get active products
        active_products = Product.query.filter_by(
            seller_id=current_user.id,
            status='available'
        ).order_by(Product.created_at.desc()).all()
        
        # Get sales data
        sales = Transaction.query.filter_by(
            seller_id=current_user.id,
            transaction_type='product'
        ).order_by(Transaction.created_at.desc()).all()
        
        # Calculate total sales
        total_sales = sum(sale.amount for sale in sales if sale.status == 'completed')
        
        # Get recent soil reports
        soil_reports = SoilReport.query.filter_by(
            farmer_id=current_user.id
        ).order_by(SoilReport.report_date.desc()).limit(5).all()
        
        # Calculate investment statistics
        total_active_amount = sum(inv.amount for inv in active_investments)
        total_completed_amount = sum(inv.amount for inv in completed_investments)
        total_returns = sum(inv.amount * (1 + inv.interest_rate/100) for inv in completed_investments)
        
        # Prepare sales chart data
        sales_labels = []
        sales_values = []
        for sale in sales:
            if sale.status == 'completed':
                sales_labels.append(sale.created_at.strftime('%Y-%m-%d'))
                sales_values.append(float(sale.amount))
        
        # Ensure all numeric values are properly formatted
        total_active_amount = float(total_active_amount) if total_active_amount else 0.0
        total_completed_amount = float(total_completed_amount) if total_completed_amount else 0.0
        total_returns = float(total_returns) if total_returns else 0.0
        total_sales = float(total_sales) if total_sales else 0.0
        
        return render_template('farmer/dashboard.html',
                            pending_requests=pending_requests,
                            active_investments=active_investments,
                            completed_investments=completed_investments,
                            pending_offers=pending_offers,
                            active_products=active_products,
                            total_sales=total_sales,
                            soil_reports=soil_reports,
                            total_active_amount=total_active_amount,
                            total_completed_amount=total_completed_amount,
                            total_returns=total_returns,
                            products=active_products,
                            investment_requests=pending_requests,
                            investment_offers=pending_offers,
                            sales_labels=sales_labels,
                            sales_values=sales_values)
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@farmer_bp.route('/farmer/products', methods=['GET', 'POST'])
@login_required
@farmer_required
def products():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            description = request.form.get('description')
            price = float(request.form.get('price'))
            quantity = float(request.form.get('quantity'))
            unit = request.form.get('unit')
            category = request.form.get('category')
            
            # Handle image upload
            image = request.files.get('image')
            image_url = None
            if image:
                filename = secure_filename(image.filename)
                image.save(os.path.join('app/static/uploads/products', filename))
                image_url = f'/static/uploads/products/{filename}'
            
            product = Product(
                name=name,
                description=description,
                price=price,
                quantity=quantity,
                unit=unit,
                category=category,
                image_url=image_url,
                seller_id=current_user.id,
                status='available'
            )
            
            db.session.add(product)
            db.session.commit()
            
            flash('Product added successfully!', 'success')
            return redirect(url_for('farmer.products'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while adding the product: {str(e)}', 'error')
            return redirect(url_for('farmer.products'))
    
    # Get all products for the current farmer
    products = Product.query.filter_by(seller_id=current_user.id).order_by(Product.created_at.desc()).all()
    return render_template('farmer/products.html', products=products)

@farmer_bp.route('/farmer/investments', methods=['GET', 'POST'])
@login_required
@farmer_required
def investments():
    if request.method == 'POST':
        try:
            # Get and validate form data
            try:
                amount = float(request.form.get('amount', 0))
                duration = int(request.form.get('duration', 0))
                interest_rate = float(request.form.get('interest_rate', 0))
            except (ValueError, TypeError):
                flash('Please enter valid numeric values for amount, duration, and interest rate.', 'error')
                return redirect(url_for('farmer.investments'))
            
            # Validate numeric values
            if amount <= 0:
                flash('Investment amount must be greater than zero.', 'error')
                return redirect(url_for('farmer.investments'))
            if duration <= 0:
                flash('Duration must be greater than zero.', 'error')
                return redirect(url_for('farmer.investments'))
            if interest_rate <= 0 or interest_rate > 20:
                flash('Interest rate must be between 0% and 20%.', 'error')
                return redirect(url_for('farmer.investments'))
            
            # Get project details
            crop_type = request.form.get('crop_type', '')
            land_size = request.form.get('land_size', '')
            expected_yield = request.form.get('expected_yield', '')
            project_title = request.form.get('project_title', '')
            description = request.form.get('description', '')
            
            # Create investment request
            description_text = f"""
Project: {project_title}

Investment Request Details:
- Crop Type: {crop_type}
- Land Size: {land_size} acres
- Expected Yield: {expected_yield}
- Additional Details: {description}

Farmer Details:
- Name: {current_user.name}
- Farm Name: {current_user.farm_name}
- Location: {current_user.farm_location}
- Experience: {current_user.farm_size} acres managed
"""
            
            investment = Investment(
                farmer_id=current_user.id,
                investor_id=None,  # Will be set when an investor makes an offer
                amount=amount,
                duration=duration,
                interest_rate=interest_rate,
                description=description_text,
                status='seeking'  # Initial status for investment requests
            )
            
            db.session.add(investment)
            db.session.commit()
            
            flash('Investment request created successfully!', 'success')
            return redirect(url_for('farmer.investments'))
            
        except Exception as e:
            db.session.rollback()
            print(f"Investment Request Error: {str(e)}")
            flash('An error occurred while creating the investment request. Please try again.', 'error')
            return redirect(url_for('farmer.investments'))
    
    # Get my investment requests
    my_requests = Investment.query.filter_by(
        farmer_id=current_user.id,
        status='seeking'  # Only show active requests
    ).order_by(Investment.created_at.desc()).all()
    
    # Get investment offers
    investment_offers = Investment.query.filter_by(
        farmer_id=current_user.id,
        status='offered'
    ).order_by(Investment.created_at.desc()).all()
    
    # Get active investments
    active_investments = Investment.query.filter_by(
        farmer_id=current_user.id,
        status='active'
    ).order_by(Investment.created_at.desc()).all()
    
    # Get completed investments
    completed_investments = Investment.query.filter_by(
        farmer_id=current_user.id,
        status='completed'
    ).order_by(Investment.created_at.desc()).all()
    
    # Calculate total statistics
    total_active_amount = sum(inv.amount for inv in active_investments)
    total_completed_amount = sum(inv.amount for inv in completed_investments)
    total_returns = sum(inv.amount * (1 + inv.interest_rate/100) for inv in completed_investments)
    
    return render_template('farmer/investments.html',
                         my_requests=my_requests,
                         investment_offers=investment_offers,
                         active_investments=active_investments,
                         completed_investments=completed_investments,
                         total_active_amount=total_active_amount,
                         total_completed_amount=total_completed_amount,
                         total_returns=total_returns)

@farmer_bp.route('/farmer/investment/<int:investment_id>/action', methods=['POST'])
@login_required
@farmer_required
def investment_action(investment_id):
    try:
        investment = Investment.query.get_or_404(investment_id)
        
        # Verify ownership
        if investment.farmer_id != current_user.id:
            flash('Unauthorized action.', 'error')
            return redirect(url_for('farmer.investments'))
        
        action = request.form.get('action')
        if action not in ['accept', 'reject', 'cancel']:
            flash('Invalid action.', 'error')
            return redirect(url_for('farmer.investments'))
        
        # Map actions to status transitions
        action_status_map = {
            'accept': 'active',
            'reject': 'rejected',
            'cancel': 'cancelled'
        }
        
        new_status = action_status_map[action]
        
        # Attempt status transition
        if investment.transition_to(new_status):
            db.session.commit()
            flash(f'Investment {action}ed successfully!', 'success')
        else:
            flash(f'Cannot {action} investment in its current status.', 'error')
        
        return redirect(url_for('farmer.investments'))
        
    except Exception as e:
        db.session.rollback()
        print(f"Investment Action Error: {str(e)}")
        flash('An error occurred while processing your request.', 'error')
        return redirect(url_for('farmer.investments'))

@farmer_bp.route('/farmer/soil-reports', methods=['GET', 'POST'])
@login_required
@farmer_required
def soil_reports():
    if request.method == 'POST':
        try:
            # Handle soil report upload
            report_file = request.files.get('report_file')
            if not report_file or not report_file.filename:
                flash('Please select a file to upload.', 'error')
                return redirect(url_for('farmer.soil_reports'))
            
            # Validate file type
            allowed_extensions = {'pdf', 'jpg', 'jpeg', 'png'}
            if '.' not in report_file.filename or \
               report_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                flash('Invalid file type. Allowed types are: PDF, JPG, JPEG, PNG', 'error')
                return redirect(url_for('farmer.soil_reports'))
            
            # Create upload directory if it doesn't exist
            upload_dir = os.path.join('app', 'static', 'uploads', 'soil_reports')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save the file
            filename = secure_filename(report_file.filename)
            file_path = os.path.join(upload_dir, filename)
            report_file.save(file_path)
            report_url = f'/static/uploads/soil_reports/{filename}'
            
            # Get form data
            try:
                ph_level = float(request.form.get('ph_level', 0))
                nitrogen_level = float(request.form.get('nitrogen_level', 0))
                phosphorus_level = float(request.form.get('phosphorus_level', 0))
                potassium_level = float(request.form.get('potassium_level', 0))
                organic_matter = float(request.form.get('organic_matter', 0))
                moisture_content = float(request.form.get('moisture_content', 0))
            except (ValueError, TypeError):
                flash('Please enter valid numerical values for all soil measurements.', 'error')
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(url_for('farmer.soil_reports'))
            
            # Create soil report
            report = SoilReport(
                farmer_id=current_user.id,
                ph_level=ph_level,
                nitrogen_level=nitrogen_level,
                phosphorus_level=phosphorus_level,
                potassium_level=potassium_level,
                organic_matter=organic_matter,
                moisture_content=moisture_content,
                report_file_url=report_url,
                notes=request.form.get('notes', '')
            )
            
            db.session.add(report)
            db.session.commit()
            
            flash('Soil report uploaded successfully!', 'success')
            return redirect(url_for('farmer.soil_reports'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while uploading the soil report: {str(e)}', 'error')
            return redirect(url_for('farmer.soil_reports'))
    
    # GET request - display soil reports
    reports = SoilReport.query.filter_by(
        farmer_id=current_user.id
    ).order_by(SoilReport.report_date.desc()).all()
    
    return render_template('farmer/soil_reports.html', reports=reports)

@farmer_bp.route('/farmer/schemes')
@login_required
@farmer_required
def schemes():
    active_schemes = GovScheme.query.filter_by(status='active').all()
    my_applications = SchemeApplication.query.filter_by(farmer_id=current_user.id).all()
    return render_template('farmer/schemes.html',
                         schemes=active_schemes,
                         applications=my_applications)

@farmer_bp.route('/farmer/apply-scheme/<int:scheme_id>', methods=['POST'])
@login_required
@farmer_required
def apply_scheme(scheme_id):
    scheme = GovScheme.query.get_or_404(scheme_id)
    
    # Check if already applied
    existing_application = SchemeApplication.query.filter_by(
        farmer_id=current_user.id,
        scheme_id=scheme_id
    ).first()
    
    if existing_application:
        flash('You have already applied for this scheme.', 'error')
        return redirect(url_for('farmer.schemes'))
    
    # Handle document upload
    documents = request.files.get('documents')
    documents_url = None
    if documents:
        filename = secure_filename(documents.filename)
        documents.save(os.path.join('app/static/uploads/scheme_docs', filename))
        documents_url = f'/static/uploads/scheme_docs/{filename}'
    
    application = SchemeApplication(
        farmer_id=current_user.id,
        scheme_id=scheme_id,
        documents_url=documents_url,
        notes=request.form.get('notes')
    )
    
    db.session.add(application)
    db.session.commit()
    
    flash('Application submitted successfully!', 'success')
    return redirect(url_for('farmer.schemes'))

@farmer_bp.route('/farmer/materials')
@login_required
@farmer_required
def materials():
    return render_template('farmer/materials.html')

@farmer_bp.route('/farmer/analytics')
@login_required
@farmer_required
def analytics():
    try:
        # Get sales data
        sales = Transaction.query.filter_by(
            seller_id=current_user.id,
            transaction_type='product'
        ).all()
        
        # Get investment data
        investments = Investment.query.filter_by(farmer_id=current_user.id).all()
        
        # Calculate investment statistics
        active_investments = [inv for inv in investments if inv.status == 'active']
        completed_investments = [inv for inv in investments if inv.status == 'completed']
        
        # Calculate total amounts
        total_active_amount = sum(inv.amount for inv in active_investments)
        total_completed_amount = sum(inv.amount for inv in completed_investments)
        
        # Calculate returns
        active_returns = sum(inv.amount * (1 + inv.interest_rate/100) for inv in active_investments)
        completed_returns = sum(inv.amount * (1 + inv.interest_rate/100) for inv in completed_investments)
        total_returns = active_returns + completed_returns
        
        # Calculate ROI percentage
        total_invested = total_active_amount + total_completed_amount
        roi_percentage = ((total_returns - total_invested) / total_invested * 100) if total_invested > 0 else 0
        
        return render_template('farmer/analytics.html',
                             sales=sales,
                             investments=investments,
                             active_investments=active_investments,
                             completed_investments=completed_investments,
                             total_active_amount=total_active_amount,
                             total_completed_amount=total_completed_amount,
                             total_returns=total_returns,
                             roi_percentage=roi_percentage)
    except Exception as e:
        print(f"Analytics Error: {str(e)}")  # Log the error
        flash('An error occurred while loading analytics. Please try again.', 'error')
        return redirect(url_for('farmer.dashboard'))

@farmer_bp.route('/farmer/products/<int:product_id>/delete', methods=['POST'])
@login_required
@farmer_required
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    
    # Ensure the product belongs to the current user
    if product.seller_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        # Delete the product image if it exists
        if product.image_url:
            image_path = os.path.join('app', 'static', product.image_url.lstrip('/'))
            if os.path.exists(image_path):
                os.remove(image_path)
        
        db.session.delete(product)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}) 

@farmer_bp.route('/farmer/crop-prediction')
@login_required
@farmer_required
def crop_prediction():
    try:
        # Get farmer's latest soil report
        latest_soil_report = SoilReport.query.filter_by(
            farmer_id=current_user.id
        ).order_by(SoilReport.report_date.desc()).first()
        
        # Get weather data (you'll need to implement this with a weather API)
        # For now, we'll use dummy data
        weather_data = {
            'temperature': 25,
            'humidity': 65,
            'rainfall': 0,
            'season': 'summer'
        }
        
        # Get market predictions (you'll need to implement this with a market API)
        # For now, we'll use dummy data
        market_predictions = {
            'rice': {'price': 45.50, 'trend': 'up'},
            'wheat': {'price': 35.75, 'trend': 'down'},
            'corn': {'price': 28.25, 'trend': 'stable'},
            'soybeans': {'price': 42.80, 'trend': 'up'}
        }
        
        # Initialize ML models
        crop_model = CropPredictionModel()
        yield_model = YieldForecastingModel()
        soil_model = SoilAnalysisModel()
        
        # Get ML-based crop recommendations
        recommended_crops = []
        soil_analysis = None
        yield_predictions = {}
        
        if latest_soil_report:
            # Convert soil report to DataFrame
            soil_df = pd.DataFrame([{
                'ph_level': latest_soil_report.ph_level,
                'nitrogen_level': latest_soil_report.nitrogen_level,
                'phosphorus_level': latest_soil_report.phosphorus_level,
                'potassium_level': latest_soil_report.potassium_level,
                'organic_matter': latest_soil_report.organic_matter,
                'moisture_content': latest_soil_report.moisture_content
            }])
            
            # Convert weather data to DataFrame
            weather_df = pd.DataFrame([{
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'rainfall': weather_data['rainfall']
            }])
            
            try:
                # Get ML-based crop recommendations
                recommended_crops = crop_model.predict_crop(soil_df, weather_df)
                
                # Get soil health analysis
                soil_analysis = soil_model.analyze_soil_health(soil_df)
                
                # Get yield predictions for recommended crops
                for crop_rec in recommended_crops[:3]:  # Top 3 recommendations
                    crop_name = crop_rec['name']
                    try:
                        yield_pred = yield_model.predict_yield(
                            soil_df, weather_df, crop_type=crop_name
                        )
                        yield_predictions[crop_name] = yield_pred
                    except Exception as e:
                        print(f"Error predicting yield for {crop_name}: {str(e)}")
                        yield_predictions[crop_name] = {
                            'predicted_yield': 0,
                            'confidence_interval': {'lower': 0, 'upper': 0},
                            'unit': 'kg/hectare'
                        }
                
            except Exception as e:
                print(f"Error in ML prediction: {str(e)}")
                # Fallback to rule-based recommendations
                recommended_crops = get_recommended_crops(latest_soil_report, weather_data)
        else:
            # No soil report available
            recommended_crops = []
            soil_analysis = {
                'health_score': 0,
                'health_category': 'unknown',
                'recommendations': [{
                    'type': 'General',
                    'priority': 'high',
                    'action': 'Upload soil report for accurate recommendations',
                    'amount': 'Required for ML-based predictions'
                }]
            }
        
        return render_template('farmer/crop_prediction.html',
                             soil_report=latest_soil_report,
                             weather_data=weather_data,
                             market_predictions=market_predictions,
                             recommended_crops=recommended_crops,
                             soil_analysis=soil_analysis,
                             yield_predictions=yield_predictions)
        
    except Exception as e:
        print(f"Error in crop prediction: {str(e)}")
        flash('An error occurred while generating crop predictions. Please try again.', 'error')
        return redirect(url_for('farmer.dashboard'))

def get_recommended_crops(soil_report, weather_data):
    # This is a simplified recommendation system
    # In a real application, you would use machine learning models
    recommendations = []
    
    if not soil_report:
        return recommendations
    
    # Example logic for crop recommendations
    if soil_report.ph_level >= 6.0 and soil_report.ph_level <= 7.5:
        if weather_data['temperature'] >= 20 and weather_data['temperature'] <= 30:
            recommendations.append({
                'name': 'Rice',
                'confidence': 85,
                'reason': 'Optimal pH and temperature conditions for rice cultivation'
            })
    
    if soil_report.nitrogen_level >= 40 and soil_report.nitrogen_level <= 60:
        if weather_data['temperature'] >= 15 and weather_data['temperature'] <= 25:
            recommendations.append({
                'name': 'Wheat',
                'confidence': 80,
                'reason': 'Good nitrogen levels and suitable temperature for wheat'
            })
    
    if soil_report.phosphorus_level >= 30 and soil_report.phosphorus_level <= 50:
        if weather_data['temperature'] >= 25 and weather_data['temperature'] <= 35:
            recommendations.append({
                'name': 'Corn',
                'confidence': 75,
                'reason': 'Adequate phosphorus and warm temperature for corn growth'
            })
    
    return recommendations

@farmer_bp.route('/farmer/agri-products')
@login_required
@farmer_required
def agri_products():
    # Get filter parameters
    category = request.args.get('category')
    sort = request.args.get('sort', 'price_low')
    search = request.args.get('search', '')
    
    # Base query for agricultural products
    query = Product.query.filter(
        Product.category.in_(['seeds', 'fertilizers', 'pesticides', 'equipment']),
        Product.status == 'available'
    )
    
    # Apply filters
    if category:
        query = query.filter(Product.category == category)
    if search:
        query = query.filter(
            (Product.name.ilike(f'%{search}%')) |
            (Product.description.ilike(f'%{search}%'))
        )
    
    # Apply sorting
    if sort == 'price_low':
        query = query.order_by(Product.price.asc())
    elif sort == 'price_high':
        query = query.order_by(Product.price.desc())
    elif sort == 'name':
        query = query.order_by(Product.name.asc())
    
    products = query.all()
    return render_template('farmer/agri_products.html', products=products)

@farmer_bp.route('/farmer/purchase-product/<int:product_id>', methods=['POST'])
@login_required
@farmer_required
def purchase_product(product_id):
    try:
        product = Product.query.get_or_404(product_id)
        
        # Validate product category
        if product.category not in ['seeds', 'fertilizers', 'pesticides', 'equipment']:
            return jsonify({
                'success': False,
                'message': 'Invalid product category.'
            })
        
        # Get form data
        quantity = float(request.form.get('quantity', 0))
        delivery_address = request.form.get('delivery_address', '').strip()
        payment_method = request.form.get('payment_method')
        
        # Validate quantity
        if quantity <= 0:
            return jsonify({
                'success': False,
                'message': 'Please enter a valid quantity.'
            })
        
        if quantity > product.quantity:
            return jsonify({
                'success': False,
                'message': 'Requested quantity exceeds available stock.'
            })
        
        # Validate delivery address
        if not delivery_address:
            return jsonify({
                'success': False,
                'message': 'Please provide a delivery address.'
            })
        
        # Calculate total amount
        total_amount = product.price * quantity
        
        # Handle different payment methods
        if payment_method == 'wallet':
            # Check if user has sufficient wallet balance
            if current_user.wallet_balance < total_amount:
                return jsonify({
                    'success': False,
                    'message': 'Insufficient wallet balance. Please add funds to your wallet.'
                })
            
            # Update user's wallet balance
            current_user.wallet_balance -= total_amount
            
            # Update seller's wallet balance
            seller = User.query.get(product.seller_id)
            seller.wallet_balance += total_amount
            
            payment_status = 'completed'
        elif payment_method == 'cod':
            # For Cash on Delivery, create a pending transaction
            payment_status = 'pending'
        else:
            # For card and UPI payments, create a pending transaction
            payment_status = 'pending'
        
        # Create transaction
        transaction = Transaction(
            buyer_id=current_user.id,
            seller_id=product.seller_id,
            product_id=product.id,
            amount=total_amount,
            quantity=quantity,
            transaction_type='product',  # Ensure this is set to 'product'
            status=payment_status,
            payment_method=payment_method,
            delivery_address=delivery_address
        )
        
        # Update product quantity
        product.quantity -= quantity
        if product.quantity == 0:
            product.status = 'sold'
        
        db.session.add(transaction)
        db.session.commit()
        
        print(f"Transaction created: {transaction.id}, Type: {transaction.transaction_type}, Status: {transaction.status}")  # Debug print
        
        return jsonify({
            'success': True,
            'message': 'Purchase completed successfully!' + (' Payment will be collected on delivery.' if payment_method == 'cod' else ''),
            'transaction_id': transaction.id,
            'total_amount': total_amount
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Purchase Error: {str(e)}")  # Debug print
        return jsonify({
            'success': False,
            'message': f'An error occurred while processing your purchase: {str(e)}'
        })

@farmer_bp.route('/farmer/dashboard/pending-requests')
@login_required
@farmer_required
def pending_requests():
    try:
        # Get all types of investments
        investment_requests = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='seeking'
        ).order_by(Investment.created_at.desc()).all()
        
        investment_offers = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='offered'
        ).order_by(Investment.created_at.desc()).all()
        
        active_investments = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='active'
        ).order_by(Investment.created_at.desc()).all()
        
        completed_investments = Investment.query.filter_by(
            farmer_id=current_user.id,
            status='completed'
        ).order_by(Investment.created_at.desc()).all()
        
        # Get active products
        products = Product.query.filter_by(
            seller_id=current_user.id,
            status='available'
        ).order_by(Product.created_at.desc()).all()
        
        # Calculate total amounts
        total_active_amount = sum(inv.amount for inv in active_investments)
        total_completed_amount = sum(inv.amount for inv in completed_investments)
        total_returns = sum(inv.amount * (1 + inv.interest_rate/100) for inv in completed_investments)
        
        # Convert to JSON-serializable format
        def investment_to_dict(investment):
            return {
                'id': investment.id,
                'description': investment.description,
                'amount': float(investment.amount),
                'duration': investment.duration,
                'status': investment.status
            }
            
        def product_to_dict(product):
            return {
                'id': product.id,
                'name': product.name,
                'price': float(product.price),
                'quantity': float(product.quantity),
                'unit': product.unit,
                'status': product.status
            }
        
        return jsonify({
            'pending_offers': len(investment_offers),
            'investment_requests': [investment_to_dict(r) for r in investment_requests[:2]],
            'investment_offers': [investment_to_dict(o) for o in investment_offers[:2]],
            'active_investments': [investment_to_dict(i) for i in active_investments[:2]],
            'total_active_amount': float(total_active_amount),
            'total_completed_amount': float(total_completed_amount),
            'total_returns': float(total_returns),
            'active_products_count': len(products),
            'products': [product_to_dict(p) for p in products[:5]]  # Get latest 5 products
        })
        
    except Exception as e:
        print(f"Pending Requests Error: {str(e)}")
        return jsonify({'error': 'Failed to fetch pending requests'}), 500 